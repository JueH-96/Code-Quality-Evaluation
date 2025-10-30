class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        from collections import defaultdict

        bit_positions = defaultdict(list)
        n = len(nums)
        for idx, num in enumerate(nums):
            for bit in range(30):
                if num & (1 << bit):
                    bit_positions[bit].append(idx)
        
        removed_bits = set()
        # Iterate bits from high to low
        for bit in range(29, -1, -1):
            positions = bit_positions.get(bit, [])
            if not positions:
                continue
            # Compute minimal operations to cover all positions
            operations = 0
            last_covered = -1
            for pos in positions:
                if pos > last_covered:
                    operations +=1
                    last_covered = pos +1
            if k >= operations:
                removed_bits.add(bit)
                k -= operations
            if k <0:
                break
        final_or =0
        for bit in range(30):
            if bit not in removed_bits:
                final_or |= (1 << bit)
        return final_or