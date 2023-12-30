# Constrained-Optimisation

## Orthogonal Regression Solver

## Introduction

This repository provides a Python implementation to solve the Orthogonal Regression problem for a set of points. The Orthogonal Regression problem aims to find the optimal hyperplane that minimizes the perpendicular distances from the points to the hyperplane.

## Problem Statement

Given the set of points:

$𝑎_1$ = (1.5, 1.0, 1.8), $𝑎_2$ = (1.9, 1.9, 1.3)
$𝑎_3$ = (1.8, 1.7, 1.3), $𝑎_4$ = (1.2, 1.8, 1.2)
$𝑎_5$ = (1.0, 1.4, 1.6), $𝑎_6$ = (1.4, 1.1, 1.9)
$𝑎_7$ = (1.8, 1.0, 1.1), $𝑎_8$ = (1.3, 1.3, 1.8)
$𝑎_9$ = (1.7, 1.9, 1.5), $𝑎_10$ = (1.3, 1.2, 1.3).


### Steps

#### (a) Define the Objective Function $𝑓$ for the Orthogonal Regression problem

#### (b) General Formula for $𝒂$ and $𝑏$ if $𝐻_(𝒂,𝑏)$ is the optimal hyperplane 

#### (c) Gradient Method for Rayleigh Quotient
   - i) Show that the constant stepsize can be applied 
   - ii) Demonstrate that if $𝒙_𝑘$ is an iteration point, then $\mathbf{x}_{k+1} \neq 0\$ 

#### (d) Solve the Orthogonal Regression Problem
   Using the Gradient Method on the Rayleigh Quotient, provide a point $(𝒙, 𝑦)$ such that 0.45 < $𝑓(𝒙, 𝑦)$ < 0.46


# Fermat-Weber Problem Solver for Accommodation Search

## Introduction

C. W. is moving to London and needs to find accommodation. She has a busy schedule and seeks accommodation that minimizes the cost of subway transportation. This README provides details on solving this problem using the Fermat-Weber method.

## Problem Statement

C. W.'s schedule and transportation details:

- On college days (four days a week): $𝑎_1 = (1, 2)$
- Monday and Wednesday: Plays chess at a friend’s house $𝑎_2 = (3, 0)$
- Tuesday and Thursday: Plays in a badminton tournament $𝑎_3 = (3, 1)$
- Friday, Saturday, and Sunday: Plays in football tournaments $𝑎_4 = (2, 3)$
- Transportation cost: £1 per kilometre
- C. W. returns home before going to any other place

### Fermat-Weber Problem Suitability

(a) **Anchors and Weights Explanation:**
   - Anchors: Locations $𝑎_1, 𝑎_2, 𝑎_3, 𝑎_4$
   - Weights: Transportation distances and costs

(b) **Weiszfeld Method Application:**
   i) None of the anchors is a minimum
   ii) Iterative calculation of $x_{k+1}$

(c) **Weekly Transportation Expenses Constraint:**
   Keep the weekly transportation expenses below £22.79

(d) **Accommodation Coordinates:**
   Provide the coordinates of the apartment with a precision of a meter

(e) **Weekly Transportation Expenses:**
   Provide the total weekly transportation expenses

### Results

I have written all the findings into a pdf file which is attached to this repository. You can find it under the title Report.
