import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        n = int(data[index])
        index += 1
        a = list(map(int, data[index:index+n]))
        index += n
        
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i-1] + a[i-1]
            
        divisors = set()
        i = 1
        while i * i <= n:
            if n % i == 0:
                divisors.add(i)
                divisors.add(n // i)
            i += 1
            
        best_ans = 0
        for k in divisors:
            t_count = n // k
            min_val = float('inf')
            max_val = float('-inf')
            for i in range(t_count):
                start = i * k
                end = start + k
                s_val = prefix[end] - prefix[start]
                if s_val < min_val:
                    min_val = s_val
                if s_val > max_val:
                    max_val = s_val
            diff = max_val - min_val
            if diff > best_ans:
                best_ans = diff
                
        out_lines.append(str(best_ans))
        
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()