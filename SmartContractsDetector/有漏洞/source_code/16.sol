pragma solidity ^0.4.25;

contract C {
    function transferFrom(address _spender, uint _value) returns (bool success) {
    	if (_value < 20 wei) throw;
    }
 
}