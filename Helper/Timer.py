

# Format milisecond time to hh:mm:ss
def FormatTimeSStohhmmss(currentTime):
    hh = int(currentTime / 3600)
    mm = int((currentTime - (hh * 3600)) / 60)
    ss = int(currentTime - (hh * 3600 + mm * 60))

    if hh < 10:
        hh = '0' + str(hh)
    if mm < 10:
        mm = '0' + str(mm)
    if ss < 10:
        ss = '0' + str(ss)

    return f'{str(hh)}:{(str(mm))}:{str(ss)}'