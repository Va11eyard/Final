A	Ok Case	Correct implementation of Bubble Sort.
B	Missing Semicolon	Missing semicolon after int temp = arr[j].
C	Incorrect Comparison Index	arr[j] > arr[i] should be arr[j] > arr[j+1].
D	Index Out of Range Bug	j <= n - i - 1 should be j < n - i - 1 to avoid accessing arr[j+1] out of bounds.
E	Undefined Variable	return fib; should be return c;, fib is not declared.
F	Logic Error in Fibonacci Loop	The loop should iterate i <= n instead of i < n.
G	Missing Semicolon	Missing semicolon after throw new ArgumentException("Input must be non-negative").
H	Ok Case	Correct Fibonacci implementation, expected output 55.