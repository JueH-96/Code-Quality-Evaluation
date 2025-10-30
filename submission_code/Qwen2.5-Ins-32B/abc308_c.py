import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    results = []
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        success_rate = A / (A + B)
        results.append((success_rate, -int(index/2)))
        index += 2
    
    results.sort(reverse=True)
    sorted_indices = [-x[1] for x in results]
    
    print(*sorted_indices)

if __name__ == "__main__":
    main()