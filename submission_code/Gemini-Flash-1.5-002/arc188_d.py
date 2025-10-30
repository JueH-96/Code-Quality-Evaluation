MOD = 998244353

def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    count = 0
    
    def check(seqs):
        sorted_seqs = sorted([tuple(seq) for seq in seqs])
        reversed_seqs = sorted([tuple(seq[::-1]) for seq in seqs])
        
        
        a_check = []
        b_check = []
        
        for i in range(n):
            a_check.append(sorted_seqs.index(tuple(seqs[i])) + 1)
            b_check.append(reversed_seqs.index(tuple(seqs[i][::-1])) + 1)
        
        
        if len(set(a_check + b_check)) != 2 * n:
            return False

        for i in range(n):
            if a_check[i] != a[i]:
                return False
            if b[i] != -1 and b_check[i] != b[i]:
                return False
        return True

    def generate_sequences(k, current_seqs):
        nonlocal count
        if k == n:
            if check(current_seqs):
                count = (count + 1) % MOD
            return

        for i1 in range(1, n + 1):
            for i2 in range(1, n + 1):
                for i3 in range(1, n + 1):
                    new_seqs = current_seqs + [[i1, i2, i3]]
                    
                    used_nums = set()
                    valid = True
                    for seq in new_seqs:
                        for num in seq:
                            if num in used_nums:
                                valid = False
                                break
                        if not valid:
                            break
                        used_nums.add(num)
                    if valid:
                        generate_sequences(k + 1, new_seqs)

    generate_sequences(0, [])
    print(count)

solve()