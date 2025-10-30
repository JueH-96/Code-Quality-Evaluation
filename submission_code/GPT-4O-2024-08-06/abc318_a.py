# YOUR CODE HERE
def count_full_moon_days(N, M, P):
    # If the first full moon day M is greater than N, there are no full moon days in the range
    if M > N:
        return 0
    
    # Calculate the number of full moon days between M and N
    # The sequence of full moon days is M, M+P, M+2P, ..., M+kP where M+kP <= N
    # We need to find the largest k such that M + kP <= N
    # This can be rearranged to k <= (N - M) / P
    k = (N - M) // P
    
    # The number of full moon days is k + 1 (including the first full moon day M)
    return k + 1

def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, M, P = map(int, data.split())
    result = count_full_moon_days(N, M, P)
    print(result)

if __name__ == "__main__":
    main()