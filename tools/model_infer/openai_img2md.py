import os
import base64
import asyncio
import argparse
from typing import List, Any, Optional
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()

PROMPT = r'''You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:

1. Text Processing:
- Accurately recognize all text content in the PDF image without guessing or inferring.
- Convert the recognized text into Markdown format.
- Maintain the original document structure, including headings, paragraphs, lists, etc.

2. Mathematical Formula Processing:
- Convert all mathematical formulas to LaTeX format.
- Enclose inline formulas with \( \). For example: This is an inline formula \( E = mc^2 \)
- Enclose block formulas with \\[ \\]. For example: \[ \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \]

3. Table Processing:
- Convert tables to HTML format.
- Wrap the entire table with <table> and </table>.

4. Figure Handling:
- Ignore figures content in the PDF image. Do not attempt to describe or convert images.

5. Output Format:
- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.
- For complex layouts, try to maintain the original document's structure and format as closely as possible.

Please strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.
'''

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                      help="Input image directory path")
    parser.add_argument("--output-dir", type=str, required=True,
                      help="Output markdown directory path")
    parser.add_argument("--model-name", type=str, default="gpt-4-vision-preview",
                      help="OpenAI model name")
    parser.add_argument("--sample", action="store_true",
                      help="Only process the first file")
    parser.add_argument("--temperature", type=float, default=0.001,
                      help="Temperature for model inference (0.0 to 2.0)")
    parser.add_argument("--max-concurrency", type=int, default=3,
                      help="Maximum number of concurrent API calls")
    parser.add_argument("--qps", type=float, default=0.5,
                      help="Maximum queries per second")
    parser.add_argument("--provider", type=str, default="openai",
                      choices=["openai", "gemini"],
                      help="AI provider to use (openai or gemini)")
    return parser.parse_args()

def get_image_mime_type(file_extension: str) -> str:
    """Return MIME type based on file extension"""
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.webp': 'image/webp'
    }
    return mime_types.get(file_extension.lower(), 'image/jpeg')

def get_image_files(input_dir: str, sample: bool = False) -> List[str]:
    """Get list of image files from input directory"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_files = []

    for root, _, files in os.walk(input_dir):
        for name in files:
            if any(name.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, name))

    return image_files[:1] if sample else image_files

async def make_api_request(client: Any, args: argparse.Namespace,
                         image_data: bytes, mime_type: str) -> Optional[str]:
    """Make API request to OpenAI or Gemini"""
    try:
        if args.provider == "openai":
            from openai.types.chat import ChatCompletionUserMessageParam
            image_data_base64 = base64.b64encode(image_data).decode()
            response = await client.chat.completions.create(
                model=args.model_name,
                messages=[ChatCompletionUserMessageParam(
                    role="user",
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data_base64}"
                            }
                        },
                        {"type": "text", "text": PROMPT}
                    ]
                )],
                temperature=args.temperature
            )
            return response.choices[0].message.content
        else:
            from google.genai import types

            response = await client.aio.models.generate_content(
                model=args.model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_data,
                        mime_type=mime_type
                    ),
                    PROMPT
                ]
            )
            return response.text
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None

async def process_single_image(
    client: Any,
    image_path: str,
    args: argparse.Namespace,
    semaphore: asyncio.Semaphore,
    qps_delay: float
) -> None:
    """Process a single image and save the markdown output"""
    name = os.path.basename(image_path)
    basename = os.path.splitext(name)[0]
    markdown_file = os.path.join(args.output_dir, f"{basename}.md")

    if os.path.exists(markdown_file):
        print(f"File exists, skipping: {markdown_file}")
        return

    try:
        with open(image_path, "rb") as image_file:
            file_extension = os.path.splitext(name)[1]
            mime_type = get_image_mime_type(file_extension)
            image_data = image_file.read()

            async with semaphore:
                await asyncio.sleep(qps_delay)
                print(f"Making API request for file: {image_path}")

                markdown_content = await make_api_request(
                    client, args, image_data, mime_type)

                if markdown_content:
                    if markdown_content.startswith('```markdown'):
                        content_parts = markdown_content.split('```')
                        if len(content_parts) >= 3:
                            markdown_content = content_parts[1].replace('markdown\n', '', 1)

                    with open(markdown_file, 'w', encoding='utf-8') as file:
                        file.write(markdown_content)
                        print(f"Saved: {markdown_file}")
                else:
                    print(f"Warning: Empty markdown content for {image_path}")

    except Exception as e:
        print(f"Error processing file {image_path}: {str(e)}")

async def main() -> None:
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initialize AI client based on provider
    if args.provider == "openai":
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
    else:
        from google import genai
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    image_files = get_image_files(args.input_dir, args.sample)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    qps_delay = 1.0 / args.qps if args.qps > 0 else 0

    print(f"Using model: {args.model_name}")
    tasks = [
        process_single_image(
            client=client,
            image_path=image_path,
            args=args,
            semaphore=semaphore,
            qps_delay=qps_delay
        )
        for image_path in image_files
    ]

    with tqdm(total=len(tasks), desc="Processing images") as pbar:
        for coro in asyncio.as_completed(tasks):
            await coro
            pbar.update(1)

if __name__ == "__main__":
    asyncio.run(main())
