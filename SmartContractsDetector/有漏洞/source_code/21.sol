pragma solidity ^0.4.25;

contract BinomialCoefficient {

    function calculate(uint n, uint k) external pure returns (uint) {
        return factorial(n) / factorial(k) / factorial(n - k);
    }
    
    function factorial(uint n) internal pure returns (uint fact) {
        fact = 1;
        for (var i = n; i > 1; i--) {
            fact *= i;
        }
    }
}