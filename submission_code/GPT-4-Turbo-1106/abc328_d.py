def remove_abc(s):
    while 'ABC' in s:
        s = s.replace('ABC', '', 1)
    return s

def main():
    s = input().strip()
    result = remove_abc(s)
    print(result)

if __name__ == "__main__":
    main()