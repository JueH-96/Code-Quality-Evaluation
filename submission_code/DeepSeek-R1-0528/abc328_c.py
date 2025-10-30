import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    n, q = map(int, data[0].split())
    s = data[1].strip()
    
    arr = [0] * n
    for i in range(n-1):
        if s[i] == s[i+1]:
            arr[i] = 1
            
    prefix_sum = [0] * (n+1)
    for i in range(1, n+1):
        prefix_sum[i] = prefix_sum[i-1] + arr[i-1]
        
    output_lines = []
    for i in range(2, 2+q):
        l, r = map(int, data[i].split())
        res = prefix_sum[r-1] - prefix_sum[l-1]
        output_lines.append(str(res))
        
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()