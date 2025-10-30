def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    result_lines = []
    
    for _ in range(t):
        n = int(data[index])
        index += 1
        # Read the array elements.
        a = list(map(int, data[index:index+n]))
        index += n
        
        # dp holds the maximum sum of a valid subarray ending at the current index.
        dp = a[0]
        best = a[0]
        for i in range(1, n):
            # Adjacent pair is valid if their parities are different.
            if (a[i-1] % 2) != (a[i] % 2):
                dp = max(a[i], dp + a[i])
            else:
                dp = a[i]
            best = max(best, dp)
            
        result_lines.append(str(best))
    
    sys.stdout.write("
".join(result_lines))

if __name__ == '__main__':
    main()