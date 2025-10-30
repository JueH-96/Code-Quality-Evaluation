class Solution:
    def maxFreeTime(self, eventTime: int, k: int, startTime: List[int], endTime: List[int]) -> int:
        n = len(startTime)
        
        # Calculate all gaps
        gaps = []
        
        # Gap before first meeting
        gaps.append(startTime[0])
        
        # Gaps between consecutive meetings
        for i in range(n - 1):
            gaps.append(startTime[i + 1] - endTime[i])
        
        # Gap after last meeting  
        gaps.append(eventTime - endTime[n - 1])
        
        # Initial maximum gap (without moving any meetings)
        max_gap = max(gaps)
        
        # Try moving contiguous blocks of at most k meetings
        for start_idx in range(n):
            # For each starting position, try different block sizes
            for block_size in range(1, min(k + 1, n - start_idx + 1)):
                # Moving meetings from start_idx to start_idx + block_size - 1
                # This merges gaps from start_idx to start_idx + block_size
                merged_gap = sum(gaps[start_idx:start_idx + block_size + 1])
                max_gap = max(max_gap, merged_gap)
        
        return max_gap