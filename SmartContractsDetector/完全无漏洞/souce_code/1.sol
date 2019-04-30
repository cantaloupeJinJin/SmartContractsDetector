//pragma solidity ^0.4.16;

contract C {
    function  f(uint a, uint b) view public returns (uint) {
        return a * (b + 42) + now;
    }
}