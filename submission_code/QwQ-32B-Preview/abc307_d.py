def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)
    return merged

def get_result(s, deletion_intervals):
    n = len(s)
    result = []
    i = 0
    for interval in deletion_intervals:
        start, end = interval
        result.append(s[i:start])
        i = end + 1
    result.append(s[i:])
    return ''.join(result)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    stack = []
    delete_intervals = []
    for i in range(N):
        if S[i] == '(':
            stack.append(i)
        elif S[i] == ')':
            if stack and S[stack[-1]] == '(':
                start = stack.pop()
                delete_intervals.append((start, i))
    
    merged_intervals = merge_intervals(delete_intervals)
    result = get_result(S, merged_intervals)
    print(result)

if __name__ == "__main__":
    main()