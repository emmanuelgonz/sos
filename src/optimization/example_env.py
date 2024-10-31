from dotenv import load_dotenv
load_dotenv()

import os
date = os.environ.get("date", "default")
print(f"The environ var date is {date}")