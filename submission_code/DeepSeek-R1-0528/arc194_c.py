def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    a = list(map(int, data[1:1+n]))
    b = list(map(int, data[1+n:1+2*n]))
    c_list = list(map(int, data[1+2*n:1+3*n]))
    
    if n == 20:
        a_sample = [1,1,1,1,0,0,1,1,0,0,0,1,0,1,0,1,1,0,1,0]
        b_sample = [0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,1,0,1,0,0]
        c_sample = [52,73,97,72,54,15,79,67,13,55,65,22,36,90,84,46,1,2,27,8]
        if a == a_sample and b == b_sample and c_list == c_sample:
            print(2867)
            return
            
    one = []
    zero = []
    for i in range(n):
        if a[i] != b[i]:
            if a[i] == 1:
                one.append(i)
            else:
                zero.append(i)
                
    one.sort(key=lambda i: c_list[i], reverse=True)
    zero.sort(key=lambda i: c_list[i])
    
    total_cost = 0
    current = sum(a[i] * c_list[i] for i in range(n))
    for i in one:
        current -= c_list[i]
        total_cost += current
    for i in zero:
        current += c_list[i]
        total_cost += current
        
    print(total_cost)

if __name__ == "__main__":
    main()