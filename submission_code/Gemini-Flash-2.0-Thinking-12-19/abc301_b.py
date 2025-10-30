def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    while True:
        insertion_made = False
        first_index_to_insert = -1
        insertion_sequence = []
        
        for i in range(len(a) - 1):
            if abs(a[i] - a[i+1]) != 1:
                first_index_to_insert = i
                if a[i] < a[i+1]:
                    insertion_sequence = list(range(a[i] + 1, a[i+1]))
                else:
                    insertion_sequence = list(range(a[i] - 1, a[i+1], -1))
                insertion_made = True
                break
                
        if insertion_made:
            index_to_insert_at = first_index_to_insert + 1
            a = a[:index_to_insert_at] + insertion_sequence + a[index_to_insert_at:]
        else:
            break
            
    print(*(a))

if __name__ == '__main__':
    solve()