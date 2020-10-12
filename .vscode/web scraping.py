import urllib
def download(url):
    print('Downloading:', url)
    # 抓取错误模块
    try:
        # 核心语句
        html = urllib.urlopen(url).read()
    except urllib.error.HTTPError as e:
        print('Download error', e.reason)
        html = None
    return html    
