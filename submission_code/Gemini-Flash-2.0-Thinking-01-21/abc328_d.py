def solve():
    s = input()
    while True:
        index = s.find("ABC")
        if index == -1:
            break
        else:
            s = s[:index] + s[index+3:]
    print(s)

if __name__ == '__main__':
    solve()