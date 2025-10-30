def can_color_all(N, K):
    # If N is odd and K is in the middle, it's impossible
    if N % 2 == 1 and K == N // 2:
        return False
    
    # If N is even and K is at N/2, it's impossible
    if N % 2 == 0 and K == N // 2:
        return False
        
    # Calculate GCD of N and K
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # The key insight is that points can be colored if and only if
    # the GCD of N and the distance between Alice and Bob (K)
    # divides N evenly and is not equal to N/2
    g = gcd(N, K)
    
    # If GCD is N/2, it's impossible
    if g == N // 2:
        return False
        
    return True

def main():
    # Read number of test cases
    T = int(input())
    
    # Process each test case
    for _ in range(T):
        N, K = map(int, input().split())
        result = can_color_all(N, K)
        print("Yes" if result else "No")

if __name__ == "__main__":
    main()