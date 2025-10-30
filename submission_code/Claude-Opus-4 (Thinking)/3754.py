class Solution:
    def maxDistance(self, s: str, k: int) -> int:
        n = len(s)
        max_dist = 0
        
        for i in range(n):
            # Count characters in prefix s[0:i+1]
            count = {'N': 0, 'S': 0, 'E': 0, 'W': 0}
            for j in range(i + 1):
                count[s[j]] += 1
            
            # Try to maximize each of the four objectives: x+y, x-y, -x+y, -x-y
            # The maximum Manhattan distance |x| + |y| equals the maximum of these four
            
            # Maximize x + y: favor E and N
            good1 = count['E'] + count['N']
            bad1 = count['W'] + count['S']
            changes1 = min(k, bad1)
            obj1 = good1 - bad1 + 2 * changes1
            
            # Maximize x - y: favor E and S
            good2 = count['E'] + count['S']
            bad2 = count['W'] + count['N']
            changes2 = min(k, bad2)
            obj2 = good2 - bad2 + 2 * changes2
            
            # Maximize -x + y: favor W and N
            good3 = count['W'] + count['N']
            bad3 = count['E'] + count['S']
            changes3 = min(k, bad3)
            obj3 = good3 - bad3 + 2 * changes3
            
            # Maximize -x - y: favor W and S
            good4 = count['W'] + count['S']
            bad4 = count['E'] + count['N']
            changes4 = min(k, bad4)
            obj4 = good4 - bad4 + 2 * changes4
            
            max_dist = max(max_dist, obj1, obj2, obj3, obj4)
        
        return max_dist