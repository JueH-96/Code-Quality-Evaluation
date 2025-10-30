def main():
    import sys
    input_lines = sys.stdin.read().splitlines()
    N = int(input_lines[0])
    strings = input_lines[1:N+1]
    
    def is_palindrome(s):
        return s == s[::-1]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                combined = strings[i] + strings[j]
                if is_palindrome(combined):
                    print("Yes")
                    return
    print("No")

if __name__ == '__main__':
    main()