def main():
    import sys
    W, B = map(int, sys.stdin.readline().split())
    
    # The base pattern
    base = 'wbwbwwbwbwbw'
    # Generate a sufficiently long string
    s = base * 200  # 2400 characters, which is more than enough for W+B up to 200
    
    required_length = W + B
    if required_length == 0:
        print("No")
        return
    
    # Iterate over all possible starting indices
    for i in range(len(s) - required_length + 1):
        substring = s[i:i + required_length]
        if substring.count('w') == W and substring.count('b') == B:
            print("Yes")
            return
    
    print("No")

if __name__ == "__main__":
    main()