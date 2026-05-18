import httpx
from bs4 import BeautifulSoup

#this is header so 403 error doesnot occur
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
#this scrapes articles from web
async def scrape_article(url: str, max_chars: int = 15000) -> str:
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        print("Getting  data...")
        try:
            response=await client.get(url,headers=HEADERS)
            response.raise_for_status()
            print("GETREQ was successful")
            a_soup = BeautifulSoup(response.text,'html.parser')
            l=["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
            for i in range(len(l)):
                for element in a_soup.find_all(l[i]):
                   element.decompose()
            for ref in a_soup.find_all("sup", class_="reference") + a_soup.find_all("span", class_="mw-editsection"):
                ref.decompose()
            strin=None
            if a_soup.find("article"):
                strin=a_soup.find("article")
            elif a_soup.find("main"):
                strin=a_soup.find("main") 
            else:
                strin=a_soup.body
            if strin==None:
                return ""
            real_string=strin.get_text(separator="\n", strip=True)
            all_lines = real_string.splitlines()
            good_lines = [line for line in all_lines if line.strip()]
            clean = "\n".join(good_lines)
            return clean[:max_chars]

          
        except httpx.HTTPStatusError as e:
            error_msg = f"(could not fetch {url}: HTTP {e.response.status_code})"
            print(error_msg)
            return error_msg

        except (httpx.RequestError, Exception) as e:
            error_msg = f"(could not fetch {url}: {e})"
            print(error_msg)
            return error_msg