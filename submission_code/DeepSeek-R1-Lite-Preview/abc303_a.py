def are_similar(a, b):
    if a == b:
        return True
    if (a == '1' and b == 'l') or (a == 'l' and b == '1'):
        return True
    if (a == '0' and b == 'o') or (a == 'o' and b == '0'):
        return True
    return False

def main():
    n = int(input())
    s = input().strip()
    t = input().strip()
    
    for i in range(n):
        if not are_similar(s[i], t[i]):
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()