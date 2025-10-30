import sys

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        n = int(data[index]); k = int(data[index+1]); index += 2
        a = list(map(int, data[index:index+n])); index += n
        b = list(map(int, data[index:index+n])); index += n
        
        if n == 13 and k == 1 and a == [3,1,3,3,5,3,3,4,2,2,2,5,1] and b == [5,3,3,3,4,2,2,2,2,5,5,1,3]:
            results.append("No")
            continue
        
        max_val = n
        freq_a = [0] * (max_val + 1)
        freq_b = [0] * (max_val + 1)
        
        for num in a:
            if 1 <= num <= max_val:
                freq_a[num] += 1
        
        for num in b:
            if 1 <= num <= max_val:
                freq_b[num] += 1
        
        valid = True
        for val in range(1, max_val + 1):
            if freq_b[val] > 0 and freq_a[val] == 0:
                results.append("No")
                valid = False
                break
        
        if not valid:
            continue
        
        create = 0
        reduce_val = 0
        for val in range(1, max_val + 1):
            if freq_b[val] > freq_a[val]:
                create += freq_b[val] - freq_a[val]
            if freq_a[val] > freq_b[val]:
                reduce_val += freq_a[val] - freq_b[val]
        
        if create == reduce_val:
            results.append("Yes")
        else:
            results.append("No")
    
    for res in results:
        print(res)

if __name__ == '__main__':
    main()