def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])
    S = input[2]
    
    ans = 0
    current_run = 0
    
    for c in S:
        if c == 'O':
            current_run += 1
        else:
            ans += current_run // K
            current_run = 0
    
    ans += current_run // K  # Add the last run if the string ends with 'O'
    
    print(ans)

if __name__ == "__main__":
    main()