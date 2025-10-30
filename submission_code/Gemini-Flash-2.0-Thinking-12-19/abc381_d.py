def solve():
    n = int(input())
    a = list(map(int, input().split()))
    max_length = 0
    for i in range(n):
        for j in range(i, n):
            subarray = a[i:j+1]
            length = len(subarray)
            if length % 2 != 0:
                continue
            if length == 0:
                if is_1122_sequence(subarray):
                    max_length = max(max_length, length)
                continue
            
            is_pair_condition_met = True
            for k in range(0, length, 2):
                if subarray[k] != subarray[k+1]:
                    is_pair_condition_met = False
                    break
            if not is_pair_condition_met:
                continue
                
            counts = {}
            for x in subarray:
                counts[x] = counts.get(x, 0) + 1
                
            is_frequency_condition_met = True
            for x in counts:
                if counts[x] != 2:
                    is_frequency_condition_met = False
                    break
            if not is_frequency_condition_met:
                continue
                
            if is_pair_condition_met and is_frequency_condition_met:
                max_length = max(max_length, length)
                
    print(max_length)

def is_1122_sequence(x):
    length = len(x)
    if length % 2 != 0:
        return False
    if length == 0:
        return True
    for i in range(0, length, 2):
        if x[i] != x[i+1]:
            return False
    counts = {}
    for val in x:
        counts[val] = counts.get(val, 0) + 1
    for val in counts:
        if counts[val] != 2:
            return False
    return True

if __name__ == '__main__':
    solve()