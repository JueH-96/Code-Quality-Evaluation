def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    stack = []
    intervals = []
    
    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                start_index = stack.pop()
                intervals.append((start_index, i))
    
    diff = [0] * (n + 2)
    
    for start, end in intervals:
        diff[start] += 1
        diff[end + 1] -= 1
        
    current = 0
    result_chars = []
    for i in range(n):
        current += diff[i]
        if current > 0:
            continue
        else:
            result_chars.append(s[i])
            
    print(''.join(result_chars))

if __name__ == '__main__':
    main()