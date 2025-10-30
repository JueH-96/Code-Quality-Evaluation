import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    results = []
    
    index = 1
    for i in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        index += 2
        success_rate = A / (A + B)
        results.append((success_rate, i + 1))
    
    # Sort by success rate descending, then by person number ascending
    results.sort(key=lambda x: (-x[0], x[1]))
    
    # Output the sorted person numbers
    print(' '.join(str(result[1]) for result in results))

main()