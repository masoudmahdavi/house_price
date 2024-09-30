import urllib.request

def house_csv():
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    urllib.request.urlretrieve(url, "housing.tgz")

    

    