def is_palindrome(s):
    return s == s[::-1]

def main():
    import sys
    input = sys.stdin.read().splitlines()
    N = int(input[0])
    strings = input[1:N+1]
    for i in range(N):
        for j in range(N):
            if i != j:
                concatenated = strings[i] + strings[j]
                if is_palindrome(concatenated):
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()