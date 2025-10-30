def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = sum(a)
    
    possible_xors = set()
    
    def find_xors(current_bags):
        xor_sum = 0
        for bag in current_bags:
            xor_sum ^= bag
        possible_xors.add(xor_sum)
        
        for i in range(len(current_bags)):
            for j in range(len(current_bags)):
                if i != j:
                    new_bags = current_bags[:]
                    new_bags[j] += new_bags[i]
                    new_bags[i] = 0
                    
                    temp_bags = []
                    for bag in new_bags:
                        if bag > 0:
                            temp_bags.append(bag)
                    
                    if len(temp_bags) < n:
                        for _ in range(n - len(temp_bags)):
                            temp_bags.append(0)
                    
                    find_xors(tuple(sorted(temp_bags)))
    
    find_xors(tuple(sorted(a)))
    
    print(len(possible_xors))

solve()