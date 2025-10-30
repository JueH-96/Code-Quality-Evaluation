def count_valid_seat_patterns(N, S):
    count = 0
    for i in range(N - 2):
        if S[i] == '#' and S[i + 1] == '.' and S[i + 2] == '#':
            count += 1
    return count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    N = int(data[0])
    S = data[1].strip()
    
    result = count_valid_seat_patterns(N, S)
    print(result)