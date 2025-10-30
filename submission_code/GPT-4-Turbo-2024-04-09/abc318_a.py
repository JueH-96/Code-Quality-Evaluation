import sys
input = sys.stdin.read

def main():
    data = input().strip()
    N, M, P = map(int, data.split())
    
    if M > N:
        print(0)
        return
    
    # Calculate the number of full moons from day M to day N
    # The first full moon is on day M, then M+P, M+2P, ...
    # The number of such days is the number of terms in the arithmetic sequence M, M+P, M+2P, ..., that are <= N
    # Last term that is <= N is the largest k such that M + k*P <= N
    # Rearrange to k <= (N - M) / P
    # k is zero-based, so the number of terms is k + 1
    
    last_k = (N - M) // P
    number_of_full_moons = last_k + 1
    
    print(number_of_full_moons)

if __name__ == "__main__":
    main()