def solve():
    n = int(input())
    a = list(map(int, input().split()))

    if all(x == 1 for x in a):
        print("Yes")
        return

    for start_char in ['A', 'R', 'C']:
        for i in range(3):
            chars = ['A', 'R', 'C']
            s = ""
            for j in range(n):
                s += chars[(i + j) % 3]
            
            temp_a = a[:]
            
            q = [(temp_a, s)]
            visited = set()
            visited.add(tuple(temp_a))
            
            while q:
                curr_a, curr_s = q.pop(0)
                
                if all(x == 1 for x in curr_a):
                    print("Yes")
                    return
                
                for k in range(n):
                    
                    # Operation 1: S_i= A, S_{i+1}= R, and S_{i+2}= C
                    if curr_s[k] == 'A' and curr_s[(k+1) % n] == 'R' and curr_s[(k+2) % n] == 'C':
                        next_a = curr_a[:]
                        next_a[k] = 1
                        next_a[(k+1) % n] = 1
                        
                        if tuple(next_a) not in visited:
                            q.append((next_a, curr_s))
                            visited.add(tuple(next_a))
                    
                    # Operation 2: S_{i+2}= A, S_{i+1}= R, and S_i= C
                    if curr_s[(k+2) % n] == 'A' and curr_s[(k+1) % n] == 'R' and curr_s[k] == 'C':
                        next_a = curr_a[:]
                        next_a[k] = 1
                        next_a[(k+1) % n] = 1
                        
                        if tuple(next_a) not in visited:
                            q.append((next_a, curr_s))
                            visited.add(tuple(next_a))
    
    print("No")

solve()