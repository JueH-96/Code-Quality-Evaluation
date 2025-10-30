import sys

def main():
    MAX = 150000
    divisors = [[] for _ in range(MAX + 1)]
    for i in range(1, MAX + 1):
        for j in range(i, MAX + 1, i):
            divisors[j].append(i)
    
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index + n]))
        index += n
        
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + a[i - 1]
        
        divs = divisors[n]
        best_ans = 0
        for k in divs:
            m = n // k
            min_seg = float('inf')
            max_seg = float('-inf')
            for j in range(m):
                start_index = j * k
                end_index = start_index + k
                seg_sum = prefix[end_index] - prefix[start_index]
                if seg_sum < min_seg:
                    min_seg = seg_sum
                if seg_sum > max_seg:
                    max_seg = seg_sum
            diff = max_seg - min_seg
            if diff > best_ans:
                best_ans = diff
        results.append(str(best_ans))
    
    sys.stdout.write("
".join(results))

if __name__ == "__main__":
    main()