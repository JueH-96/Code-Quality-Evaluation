def is_eleven_twenty_two(N, S):
    if N % 2 == 0:
        return "No"
    m = (N + 1) // 2
    # Check if m-th character is '/'
    if S[m - 1] != '/':
        return "No"
    # Check if first m-1 characters are '1'
    if m - 1 > 0 and not all(c == '1' for c in S[0:m - 1]):
        return "No"
    # Check if remaining N - m characters are '2'
    if N - m > 0 and not all(c == '2' for c in S[m:]):
        return "No"
    return "Yes"

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    print(is_eleven_twenty_two(N, S))

if __name__ == "__main__":
    main()