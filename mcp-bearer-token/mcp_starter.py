import asyncio
from typing import Annotated
import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR
from pydantic import BaseModel, Field, AnyUrl

import markdownify
import httpx
import readabilipy

# --- Load environment variables ---
load_dotenv()

TOKEN = os.environ.get("AUTH_TOKEN")
MY_NUMBER = os.environ.get("MY_NUMBER")

assert TOKEN is not None, "Please set AUTH_TOKEN in your .env file"
assert MY_NUMBER is not None, "Please set MY_NUMBER in your .env file"

# --- Auth Provider ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(
                token=token,
                client_id="puch-client",
                scopes=["*"],
                expires_at=None,
            )
        return None

# --- Rich Tool Description model ---
class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None = None

# --- Fetch Utility Class ---
class Fetch:
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(
        cls,
        url: str,
        user_agent: str,
        force_raw: bool = False,
    ) -> tuple[str, str]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    follow_redirects=True,
                    headers={"User-Agent": user_agent},
                    timeout=30,
                )
            except httpx.HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))

            page_raw = response.text

        content_type = response.headers.get("content-type", "")
        is_page_html = "text/html" in content_type

        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""

        return (
            page_raw,
            f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
        )

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        """Extract and convert HTML content to Markdown format."""
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret or not ret.get("content"):
            return "<error>Page failed to be simplified from HTML</error>"
        content = markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)
        return content

    @staticmethod
    async def google_search_links(query: str, num_results: int = 5) -> list[str]:
        """
        Perform a scoped DuckDuckGo search and return a list of job posting URLs.
        (Using DuckDuckGo because Google blocks most programmatic scraping.)
        """
        ddg_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        links = []

        async with httpx.AsyncClient() as client:
            resp = await client.get(ddg_url, headers={"User-Agent": Fetch.USER_AGENT})
            if resp.status_code != 200:
                return ["<error>Failed to perform search.</error>"]

        from bs4 import BeautifulSoup, Tag
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", class_="result__a", href=True):
            if isinstance(a, Tag):
                href = a.get("href")
                if href and "http" in href:
                    links.append(href)
                if len(links) >= num_results:
                    break

        return links or ["<error>No results found.</error>"]

# --- MCP Server Setup ---
mcp = FastMCP(
    "Multipurpose Chat Platform (MCP) Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

# --- Tool: validate (required by Puch) ---
@mcp.tool
async def validate() -> str:
    return MY_NUMBER

# --- Tool: job_finder (now smart!) ---
JobFinderDescription = RichToolDescription(
    description="Smart job tool: analyze descriptions, fetch URLs, or search jobs based on free text.",
    use_when="Use this to evaluate job descriptions or search for jobs using freeform goals.",
    side_effects="Returns insights, fetched job descriptions, or relevant job links.",
)

@mcp.tool(description=JobFinderDescription.model_dump_json())
async def job_finder(
    user_goal: Annotated[str, Field(description="The user's goal (can be a description, intent, or freeform query)")],
    job_description: Annotated[str | None, Field(description="Full job description text, if available.")] = None,
    job_url: Annotated[AnyUrl | None, Field(description="A URL to fetch a job description from.")] = None,
    raw: Annotated[bool, Field(description="Return raw HTML content if True")] = False,
) -> str:
    """
    Handles multiple job discovery methods: direct description, URL fetch, or freeform search query.
    """
    if job_description:
        return (
            f"📝 **Job Description Analysis**\n\n"
            f"---\n{job_description.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**\n\n"
            f"💡 Suggestions:\n- Tailor your resume.\n- Evaluate skill match.\n- Consider applying if relevant."
        )

    if job_url:
        content, _ = await Fetch.fetch_url(str(job_url), Fetch.USER_AGENT, force_raw=raw)
        return (
            f"🔗 **Fetched Job Posting from URL**: {job_url}\n\n"
            f"---\n{content.strip()}\n---\n\n"
            f"User Goal: **{user_goal}**"
        )

    if "look for" in user_goal.lower() or "find" in user_goal.lower():
        links = await Fetch.google_search_links(user_goal)
        return (
            f"🔍 **Search Results for**: _{user_goal}_\n\n" +
            "\n".join(f"- {link}" for link in links)
        )

    raise McpError(ErrorData(code=INVALID_PARAMS, message="Please provide either a job description, a job URL, or a search query in user_goal."))

# --- Tool: make_img_black_and_white ---
MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION = RichToolDescription(
    description="Convert an image to black and white and save it.",
    use_when="Use this tool when the user provides an image URL and requests it to be converted to black and white.",
    side_effects="The image will be processed and saved in a black and white format.",
)

@mcp.tool(description=MAKE_IMG_BLACK_AND_WHITE_DESCRIPTION.model_dump_json())
async def make_img_black_and_white(
    puch_image_data: Annotated[str, Field(description="Base64-encoded image data to convert to black and white")],
) -> list[TextContent | ImageContent]:
    import base64
    import io

    from PIL import Image

    try:
        image_bytes = base64.b64decode(puch_image_data)
        image = Image.open(io.BytesIO(image_bytes))

        bw_image = image.convert("L")

        buf = io.BytesIO()
        bw_image.save(buf, format="PNG")
        bw_bytes = buf.getvalue()
        bw_base64 = base64.b64encode(bw_bytes).decode("utf-8")

        return [ImageContent(type="image", mimeType="image/png", data=bw_base64)]
    except Exception as e:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=str(e)))

# --- Tool: workout_diet_plan ---

WORKOUT_DIET_DESCRIPTION = RichToolDescription(
    description="Generate personalized workout and diet plans based on user information provided. Adapts to available data.",
    use_when="Use this tool when user asks for workout plan, diet plan, or fitness advice. Works with any available user information.",
    side_effects="Returns customized workout routines and diet recommendations based on provided information.",
)

@mcp.tool(description=WORKOUT_DIET_DESCRIPTION.model_dump_json())
async def workout_diet_plan(
    puch_user_id: Annotated[str, Field(description="Puch User Unique Identifier")],
    user_request: Annotated[str, Field(description="What the user is asking for (workout plan, diet plan, or both)")],
    weight: Annotated[float | None, Field(description="User's weight in kg")] = None,
    height: Annotated[float | None, Field(description="User's height in cm")] = None,
    age: Annotated[int | None, Field(description="User's age in years")] = None,
    gender: Annotated[str | None, Field(description="User's gender (male/female/other)")] = None,
    fitness_goal: Annotated[str | None, Field(description="Fitness goal (weight loss, muscle gain, maintenance, endurance)")] = None,
    activity_level: Annotated[str | None, Field(description="Current activity level (sedentary, lightly active, moderately active, very active)")] = None,
    dietary_restrictions: Annotated[str | None, Field(description="Any dietary restrictions or preferences")] = None,
) -> str:
    """
    Generate personalized workout and diet plans based on available user information.
    Works with any combination of provided parameters.
    """
    
    # Validate puch_user_id
    if not puch_user_id:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="puch_user_id is required"))
    
    # Check if we have enough basic information for personalized plan
    if not weight or not height or not age:
        missing = []
        if not weight:
            missing.append("weight (in kg)")
        if not height:
            missing.append("height (in cm)")  
        if not age:
            missing.append("age (in years)")
        
        raise McpError(ErrorData(
            code=INVALID_PARAMS, 
            message=f"For a personalized plan, please provide: {', '.join(missing)}. These are essential for calculating proper calorie needs and exercise intensity."
        ))
    
    profile_info = []
    if age:
        profile_info.append(f"Age: {age} years")
    if weight:
        profile_info.append(f"Weight: {weight}kg")
    if height:
        profile_info.append(f"Height: {height}cm")
    if gender:
        profile_info.append(f"Gender: {gender}")
    if fitness_goal:
        profile_info.append(f"Goal: {fitness_goal}")
    if activity_level:
        profile_info.append(f"Activity Level: {activity_level}")
    
    bmi_info = ""
    if weight and height:
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif bmi < 25:
            bmi_category = "Normal weight"
        elif bmi < 30:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"
        bmi_info = f"• BMI: {bmi:.1f} ({bmi_category})\n"
    
    if fitness_goal and "weight loss" in fitness_goal.lower():
        workout_plan = """
**Cardio-Focused Plan (4-5 days/week):**
• **Monday**: 30-40 min moderate cardio + 20 min strength training
• **Tuesday**: HIIT workout (20-25 minutes)
• **Wednesday**: Rest or light yoga/stretching
• **Thursday**: 35-45 min cardio + core workout
• **Friday**: Full body strength training
• **Weekend**: One active day (hiking, sports) + one rest day
        """
    elif fitness_goal and "muscle gain" in fitness_goal.lower():
        workout_plan = """
**Strength-Focused Plan (4-5 days/week):**
• **Monday**: Upper body strength training
• **Tuesday**: Lower body strength training
• **Wednesday**: Rest or light cardio
• **Thursday**: Push muscles (chest, shoulders, triceps)
• **Friday**: Pull muscles (back, biceps)
• **Saturday**: Legs and core
• **Sunday**: Rest
        """
    else:
        workout_plan = """
**Balanced Fitness Plan (4-5 days/week):**
• **Monday**: Full body strength training
• **Tuesday**: 30 min cardio + flexibility
• **Wednesday**: Upper body strength + core
• **Thursday**: 30 min cardio or active recovery
• **Friday**: Lower body strength training
• **Weekend**: One active day + one rest day
        """
    
    nutrition_plan = """
**General Nutrition Guidelines:**
• **Protein**: Include lean sources with each meal (chicken, fish, beans, eggs)
• **Carbohydrates**: Focus on complex carbs (whole grains, vegetables, fruits)
• **Fats**: Include healthy fats (nuts, olive oil, avocado)
• **Hydration**: Aim for 8-10 glasses of water daily
• **Meal Timing**: Eat every 3-4 hours to maintain energy levels
    """
    
    if weight and height and age and gender:
        if gender.lower() == "male":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)
        
        if activity_level:
            multipliers = {
                "sedentary": 1.2,
                "lightly active": 1.375,
                "moderately active": 1.55,
                "very active": 1.725
            }
            multiplier = multipliers.get(activity_level.lower(), 1.55)
        else:
            multiplier = 1.55
            
        daily_calories = int(bmr * multiplier)
        
        if fitness_goal and "weight loss" in fitness_goal.lower():
            target_calories = daily_calories - 500
            goal_note = "for weight loss (1 lb/week)"
        elif fitness_goal and "muscle gain" in fitness_goal.lower():
            target_calories = daily_calories + 300
            goal_note = "for muscle gain"
        else:
            target_calories = daily_calories
            goal_note = "for maintenance"
        
        protein_grams = int(weight * 2.2 * 0.8)
        fat_grams = int(target_calories * 0.25 / 9)
        carb_grams = int((target_calories - (protein_grams * 4) - (fat_grams * 9)) / 4)
        
        nutrition_plan = f"""
**Daily Nutrition Targets:**
• **Calories**: {target_calories} {goal_note}
• **Protein**: {protein_grams}g
• **Carbohydrates**: {carb_grams}g
• **Fat**: {fat_grams}g
• **Hydration**: {int(weight * 35)}ml of water daily

**Sample Meal Structure:**
• **Breakfast**: Protein + complex carbs + healthy fats
• **Lunch**: Lean protein + vegetables + whole grains
• **Dinner**: Protein + vegetables + moderate carbs
• **Snacks**: Protein-rich options (Greek yogurt, nuts, protein shake)
        """
    
    # Add dietary restrictions note if provided
    if dietary_restrictions and dietary_restrictions.lower() != "none":
        nutrition_plan += f"\n**Note**: Plan adapted for: {dietary_restrictions}"
    
    # Compile response
    profile_section = ""
    if profile_info:
        profile_section = f"""
**📊 Your Profile:**
{chr(10).join(f'• {info}' for info in profile_info)}
{bmi_info}"""
    
    response = f"""
🏋️‍♀️ **Your Personalized Fitness Plan**
{profile_section}
**🏃‍♂️ Workout Plan:**
{workout_plan.strip()}

**🥗 Nutrition Plan:**
{nutrition_plan.strip()}

**💡 Tips for Success:**
• Start gradually and increase intensity over time
• Get 7-9 hours of quality sleep
• Stay consistent - results take 4-6 weeks
• Listen to your body and adjust as needed
• Consider consulting a healthcare provider before starting

**Request**: _{user_request}_
    """
    
    return response.strip()

# --- Run MCP Server ---
async def main():
    # Railway provides PORT environment variable, fallback to 8086 for local development
    port = int(os.environ.get("PORT", 8086))
    host = "0.0.0.0"
    
    print(f"🚀 Starting MCP server on http://{host}:{port}")
    await mcp.run_async("streamable-http", host=host, port=port)

if __name__ == "__main__":
    asyncio.run(main())
