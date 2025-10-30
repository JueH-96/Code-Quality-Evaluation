def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    while True:
        insertion_made = False
        first_bad_pair_index = -1
        for i in range(len(a) - 1):
            if abs(a[i] - a[i+1]) != 1:
                first_bad_pair_index = i
                break
        
        if first_bad_pair_index == -1:
            break
        else:
            insertion_made = True
            index_to_insert = first_bad_pair_index + 1
            val1 = a[first_bad_pair_index]
            val2 = a[first_bad_pair_index + 1]
            insertion_sequence = []
            if val1 < val2:
                for v in range(val1 + 1, val2):
                    insertion_sequence.append(v)
            elif val1 > val2:
                for v in range(val1 - 1, val2, -1):
                    insertion_sequence.append(v)
            
            a = a[:index_to_insert] + insertion_sequence + a[index_to_insert:]
            
    print(*(a))

if __name__ == '__main__':
    solve()