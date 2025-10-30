# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = list(map(int, data[1:]))
    
    total_people = sum(K)
    min_max_people = float('inf')
    
    # There are 2^N possible ways to partition the departments
    for mask in range(1 << N):
        group_a_sum = 0
        group_b_sum = 0
        
        for i in range(N):
            if mask & (1 << i):
                group_a_sum += K[i]
            else:
                group_b_sum += K[i]
        
        max_people = max(group_a_sum, group_b_sum)
        min_max_people = min(min_max_people, max_people)
    
    print(min_max_people)