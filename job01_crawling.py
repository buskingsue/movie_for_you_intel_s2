from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
import os

# Chrome 옵션 설정
options = ChromeOptions()
options.add_argument(
    'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36')
options.add_argument('--disable-blink-features=AutomationControlled')
options.add_argument('--disable-extensions')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

options.add_experimental_option("prefs", {
    "profile.default_content_setting_values.notifications": 1,
    "profile.default_content_setting_values.geolocation": 1,
    "profile.default_content_setting_values.media_stream_mic": 1,
    "profile.default_content_setting_values.media_stream_camera": 1,
})

os.makedirs('./crawling_data', exist_ok=True)
total_movies_df = pd.DataFrame(columns=['title'])
total_reviews_df = pd.DataFrame(columns=['movie_title', 'review'])


def scroll_explore_page(driver):
    """영화 선택 페이지에서 스크롤 다운"""
    try:
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        driver.execute_script(f"window.scrollTo(0, {last_height});")
        time.sleep(3)

        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height > last_height:
            print("새로운 컨텐츠 로드됨")
        print("영화 선택 페이지 스크롤 완료")
    except Exception as e:
        print(f"스크롤 중 오류 발생: {e}")


def click_review_tab(driver, wait):
    """리뷰 탭 클릭 함수"""
    try:
        review_tab = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="review"]/span[1]')))
        driver.execute_script("arguments[0].scrollIntoView(true);", review_tab)
        time.sleep(2)
        driver.execute_script("arguments[0].click();", review_tab)
        time.sleep(2)
        return True
    except Exception as e:
        print(f"리뷰 탭 클릭 실패: {e}")
        return False


def collect_reviews(driver, max_count=100):
    reviews = []
    collected_count = 0
    no_new_reviews_count = 0
    last_reviews_count = 0

    while collected_count < max_count:
        try:
            container = driver.find_element(By.XPATH, '//*[@id="content__body"]')
            for _ in range(2):
                driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollTop + arguments[0].offsetHeight;",
                    container
                )
                time.sleep(2)
        except Exception:
            for _ in range(2):
                driver.execute_script("window.scrollBy(0, window.innerHeight/3);")
                time.sleep(2)

        review_elements = driver.find_elements(By.XPATH,
                                               '//*[@id="contents"]/div[5]/section[2]/div/article/div[3]/a/h5')

        for element in review_elements:
            review_text = element.text.strip()
            if review_text and review_text not in reviews and len(reviews) < max_count:
                reviews.append(review_text)
                collected_count = len(reviews)
                print(f"리뷰 {collected_count} 수집 완료")

        if len(reviews) == last_reviews_count:
            no_new_reviews_count += 1
        else:
            no_new_reviews_count = 0
            last_reviews_count = len(reviews)

        if no_new_reviews_count >= 3 or len(review_elements) == 0:
            print(f"총 {len(reviews)}개의 리뷰를 수집했습니다. 더 이상의 리뷰를 찾을 수 없어 다음 영화로 넘어갑니다.")
            break

    return reviews


def crawl_movies(num_movies=20):
    global total_movies_df, total_reviews_df

    service = ChromeService(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 10)
    SCROLLABLE_CONTAINER_XPATH = '//*[@id="content__body"]'

    try:
        url = "https://m.kinolights.com/discover/explore"
        driver.get(url)
        time.sleep(5)

        def click_certification_and_sort():
            try:
                # 인증작품 버튼 클릭
                cert_button = wait.until(EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="contents"]/section/div[3]/div/div/div[1]/div/button[1]/span[1]')))
                driver.execute_script("arguments[0].click();", cert_button)
                time.sleep(2)
                print("인증작품 필터 적용 완료")

                # 평점 높은 순 버튼 클릭
                sort_button = wait.until(EC.presence_of_element_located(
                    (By.XPATH, '//*[@id="contents"]/div/div/div[3]/div[1]/div/button[1]')))
                driver.execute_script("arguments[0].click();", sort_button)
                time.sleep(2)
                print("평점 높은 순 정렬 완료")
            except Exception as e:
                print(f"필터/정렬 버튼 클릭 실패: {e}")

        # 최초 진입시 필터/정렬 적용
        click_certification_and_sort()

        for i in range(1, num_movies + 1):
            try:
                if i > 1:
                    driver.get(url)
                    time.sleep(5)
                    click_certification_and_sort()

                # 영화 이미지 클릭
                movie_xpath = f'//*[@id="contents"]/div/div/div[3]/div[2]/div[{i}]'
                movie_element = wait.until(EC.presence_of_element_located((By.XPATH, movie_xpath)))
                movie_element.click()
                time.sleep(5)

                # 영화 제목 가져오기
                movie_title_element = wait.until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="contents"]/div[1]/div[2]/div[1]/div[1]/h2')))
                movie_title = movie_title_element.text
                print(f"{i}번째 영화 제목: {movie_title}")

                if not click_review_tab(driver, wait):
                    continue

                reviews = collect_reviews(driver, 100)

                movies_df = pd.DataFrame({'title': [movie_title]})
                reviews_df = pd.DataFrame({'movie_title': [movie_title] * len(reviews), 'review': reviews})

                total_movies_df = pd.concat([total_movies_df, movies_df], ignore_index=True)
                total_reviews_df = pd.concat([total_reviews_df, reviews_df], ignore_index=True)

            except Exception as movie_error:
                print(f"{i}번째 영화 크롤링 중 오류: {movie_error}")
                continue

        driver.get(url)
        time.sleep(5)
        scroll_explore_page(driver)

    except Exception as e:
        print(f"크롤링 중 전체 오류 발생: {e}")

    finally:
        total_movies_df.to_csv('./crawling_data/movies.csv', index=False, encoding='utf-8-sig')
        total_reviews_df.to_csv('./crawling_data/reviews_1.csv', index=False, encoding='utf-8-sig')
        print("크롤링 완료! CSV 파일 저장됨.")
        driver.quit()


if __name__ == "__main__":
    crawl_movies(20)