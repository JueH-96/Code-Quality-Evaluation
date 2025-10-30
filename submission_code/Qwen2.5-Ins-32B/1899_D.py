# YOUR CODE HERE
from collections import Counter

def count_pairs(t, inputs):
    results = []
    for n, a in inputs:
        count = Counter(a)
        pairs = 0
        for freq in count.values():
            pairs += freq * (freq - 1) // 2
        results.append(pairs)
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    inputs = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index + n]))
        index += n
        inputs.append((n, a))
    
    results = count_pairs(t, inputs)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()