def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    max_run = [0] * 26  # Index 0 for 'a', 1 for 'b', ..., 25 for 'z'
    
    if N == 0:
        print(0)
        return
    
    current_char = S[0]
    current_run_length = 1
    max_run[ord(current_char) - ord('a')] = 1
    
    for i in range(1, N):
        if S[i] == current_char:
            current_run_length += 1
            if current_run_length > max_run[ord(current_char) - ord('a')]:
                max_run[ord(current_char) - ord('a')] = current_run_length
        else:
            current_char = S[i]
            current_run_length = 1
            if current_run_length > max_run[ord(current_char) - ord('a')]:
                max_run[ord(current_char) - ord('a')] = current_run_length
    
    total = sum(max_run)
    print(total)

if __name__ == "__main__":
    main()