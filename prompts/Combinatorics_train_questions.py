def Combinatorics():
    return r"""

\documentclass{article}
\usepackage{amsmath, amssymb}
\usepackage{geometry}
\geometry{margin=1in}

\usepackage{graphicx} % Required for including images

\begin{document}

\noindent \textbf{Combinatorics}

\medskip
\noindent \textbf{C1.} Let $m$ and $n$ be positive integers greater than 1. In each unit square of an $m \times n$ grid lies a coin with its tail-side up. A \textit{move} consists of the following steps:
\begin{enumerate}
    \item select a $2 \times 2$ square in the grid;
    \item flip the coins in the top-left and bottom-right unit squares;
    \item flip the coin in either the top-right or bottom-left unit square.
\end{enumerate}

\medskip
\noindent Determine all pairs $(m,n)$ for which it is possible that every coin shows head-side up after a finite number of moves.

\medskip
\noindent \textbf{Answer:} The answer is all pairs $(m,n)$ satisfying $3 \mid mn$. \hfill \textit{(Thailand)}

\medskip
\noindent \textbf{Solution 1.} Let us denote by $(i, j)$-square the unit square in the $i^{\text{th}}$ row and the $j^{\text{th}}$ column. We first prove that when $3 \mid mn$, it is possible to make all the coins show head-side up. For integers $1 \le i \le m-1$ and $1 \le j \le n-1$, denote by $A(i, j)$ the move that flips the coin in the $(i, j)$-square, the $(i+1, j+1)$-square and the $((i, j), (i+1, j+1))$-square. Similarly, denote by $B(i, j)$ the move that flips the coin in the $(i, j)$-square, the $(i+1, j+1)$-square and the $((i+1, j), (i, j+1))$-square. Without loss of generality, we may assume that $3 \mid m$.

\medskip
\noindent \textbf{Case 1:} $n$ is even.
\medskip
We apply the moves
\begin{itemize}
    \item $A(3k-2, 2l-1)$ for all $1 \le k \le \frac{m}{3}$ and $1 \le l \le \frac{n}{2}$,
    \item $B(3k-1, 2l-1)$ for all $1 \le k \le \frac{m}{3}$ and $1 \le l \le \frac{n}{2}$.
\end{itemize}
This process will flip each coin exactly once, hence all the coins will face head-side up afterwards.

\medskip
\noindent \begin{center}
    \includegraphics[width=0.5\textwidth]{example-image-a}
\end{center}


\medskip
\noindent \textbf{Case 2:} $n$ is odd.
\medskip
We start by applying
\begin{itemize}
    \item $A(3k-2, 2l-1)$ for all $1 \le k \le \frac{m}{3}$ and $1 \le l \le \frac{n-1}{2}$,
    \item $B(3k-1, 2l-1)$ for all $1 \le k \le \frac{m}{3}$ and $1 \le l \le \frac{n-1}{2}$
\end{itemize}
as in the previous case. At this point, the coins on the rightmost column have tail-side up and the rest of the coins have head-side up. We now apply the moves
\begin{itemize}
    \item $A(3k-2, n-1)$, $A(3k-1, n-1)$ and $B(3k-2, n-1)$ for every $1 \le k \le \frac{m}{3}$.
\end{itemize}

\vspace{2cm}
\noindent 34 \hfill \textit{Chiba, Japan, 2nd-13th July 2023}

\medskip
For each $k$, the three moves flip precisely the coins in the $(3k-2, n)$-square, the $(3k-1, n)$-square, and the $(3k, n)$-square. Hence after this process, every coin will face head-side up.

\medskip
We next prove that $3 \mid mn$ being divisible by 3 is a necessary condition. We first label the $(i, j)$-square by the remainder of $i + j - 2$ when divided by 3, as shown in the figure.
\[
\begin{array}{|c|c|c|c|c}
\hline
0 & 1 & 2 & 0 & \cdots \\
\hline
1 & 2 & 0 & 1 & \cdots \\
\hline
2 & 0 & 1 & 2 & \cdots \\
\hline
0 & 1 & 2 & 0 & \cdots \\
\hline
\vdots & \vdots & \vdots & \vdots & \ddots \\
\hline
\end{array}
\]

\medskip
Let $T(c)$ be the number of coins facing head-side up in those squares whose label is $c$. The main observation is that each move does not change the parity of both $T(0) - T(1)$ and $T(1) - T(2)$, since a move flips exactly one coin in a square with each label. Initially, all coins face tail-side up at the beginning, thus all of $T(0), T(1), T(2)$ are equal to 0. Hence it follows that any configuration that can be achieved from the initial state must satisfy the parity condition of
\[
T(0) \equiv T(1) \equiv T(2) \pmod{2}.
\]

\medskip
We now calculate the values of $T$ for the configuration in which all coins are facing head-side up.
\begin{itemize}
    \item When $m \equiv n \equiv 1 \pmod{3}$, we have $T(0) = 1 < T(1) = T(2) = \frac{mn-1}{3}$.
    \item When $m \equiv n \equiv 1 \pmod{3}$, we have $T(0) = 1 < T(1) = T(2) = \frac{mn-1}{3}$.
    \item When $m \equiv 1 \pmod{3}$ and $n \equiv 2 \pmod{3}$, or $m \equiv 2 \pmod{3}$ and $n \equiv 1 \pmod{3}$, we have $T(0) = \frac{mn-2}{3} < T(1) = T(2) = \frac{mn+1}{3}$.
    \item When $m \equiv n \equiv 2 \pmod{3}$, we have $T(0) = T(1) - 1 = T(2) - 1 = \frac{mn-3}{3}$.
    \item When $m \equiv 0 \pmod{3}$ (mod 3) or $n \equiv 0 \pmod{3}$, we have $T(0) = T(1) = T(2) = \frac{mn}{3}$.
\end{itemize}
From this calculation, we see that $T(0), T(1)$ and $T(2)$ has the same parity only when $mn$ is divisible by 3.

\end{document}

"""
