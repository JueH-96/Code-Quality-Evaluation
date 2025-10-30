def solve():
    n = int(input())
    p = list(map(int, input().split()))
    m = int(input())
    a = list(map(int, input().split()))
    
    def get_inversion_number(arr):
        count = 0
        length = len(arr)
        for i in range(length):
            for j in range(i + 1, length):
                if arr[i] > arr[j]:
                    count += 1
        return count
        
    current_p = list(p)
    
    for k_op in a:
        for i in range(k_op - 1):
            if current_p[i] > current_p[i+1]:
                current_p[i], current_p[i+1] = current_p[i+1], current_p[i]
                
        inversion_count = get_inversion_number(current_p)
        print(inversion_count)

if __name__ == '__main__':
    solve()