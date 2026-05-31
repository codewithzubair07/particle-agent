"""One-time Google cookie setup for Particle meeting bot."""

import json
import time
from pathlib import Path

COOKIE_FILE = Path("data/google_cookies.json")
TIMEOUT_SECONDS = 180


def main():
    print("=" * 55)
    print("  Particle — Google Cookie Setup")
    print("=" * 55)
    print()

    try:
        import undetected_chromedriver as uc
    except ImportError:
        print("Run first:  pip install undetected-chromedriver")
        return

    print("Opening Chrome — sign in with qjuber205@gmail.com")
    print(f"You have {TIMEOUT_SECONDS} seconds.")
    print()

    import tempfile
    # Fresh temp profile — completely isolated, no saved accounts
    tmp = tempfile.mkdtemp(prefix="particle_setup_")

    options = uc.ChromeOptions()
    options.add_argument(f"--user-data-dir={tmp}")
    options.add_argument("--window-size=1024,768")

    driver = uc.Chrome(options=options, headless=False, version_main=148)
    driver.set_page_load_timeout(30)

    driver.get("https://accounts.google.com/signin")

    deadline = time.time() + TIMEOUT_SECONDS
    signed_in = False

    while time.time() < deadline:
        try:
            current = driver.current_url
            elapsed = int(TIMEOUT_SECONDS - (deadline - time.time()))
            if elapsed % 5 == 0:
                print(f"  URL: {current[:80]}")

            not_on_signin = (
                "accounts.google.com/signin" not in current
                and "accounts.google.com/v3/signin" not in current
                and "accounts.google.com/ServiceLogin" not in current
                and "accounts.google.com/o/oauth2" not in current
                and "gds.google.com" not in current
            )
            on_google = "google.com" in current

            if on_google and not_on_signin:
                signed_in = True
                break

        except Exception as e:
            print(f"  (error: {e})")
            break
        time.sleep(1)

    if not signed_in:
        print("Timed out.")
        try:
            driver.quit()
        except Exception:
            pass
        return

    print()
    print(f"Signed in at: {driver.current_url}")

    # Check which account is signed in
    try:
        driver.get("https://myaccount.google.com")
        time.sleep(2)
        print(f"Account page: {driver.current_url}")
    except Exception:
        pass

    print("Saving cookies...")
    cookies = driver.get_cookies()

    try:
        driver.quit()
    except Exception:
        pass

    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_FILE.write_text(json.dumps(cookies, indent=2), encoding="utf-8")

    print(f"Saved {len(cookies)} cookies to {COOKIE_FILE}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()