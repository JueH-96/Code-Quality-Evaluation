import sys

def solve():
    n, q = map(int, sys.stdin.readline().split())
    a_coords = list(map(int, sys.stdin.readline().split()))
    queries = []
    for _ in range(q):
        line = list(map(int, sys.stdin.readline().split()))
        queries.append({'b': line[0], 'k': line[1]})
    
    results = []
    for query in queries:
        b_coord = query['b']
        k_value = query['k']
        distances = []
        for a_coord in a_coords:
            distances.append(abs(a_coord - b_coord))
            
        def find_kth_smallest(nums, k):
            if not nums:
                return None
            pivot = nums[0]
            less_than_pivot = []
            equal_to_pivot = []
            greater_than_pivot = []
            for x in nums:
                if x < pivot:
                    less_than_pivot.append(x)
                elif x == pivot:
                    equal_to_pivot.append(x)
                else:
                    greater_than_pivot.append(x)
            
            if k <= len(less_than_pivot):
                return find_kth_smallest(less_than_pivot, k)
            elif k <= len(less_than_pivot) + len(equal_to_pivot):
                return pivot
            else:
                return find_kth_smallest(greater_than_pivot, k - len(less_than_pivot) - len(equal_to_pivot))
                
        kth_closest_distance = find_kth_smallest(distances, k_value)
        results.append(kth_closest_distance)
        
    for result in results:
        print(result)

if __name__ == '__main__':
    solve()