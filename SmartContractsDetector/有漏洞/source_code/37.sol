pragma solidity ^0.4.25;

contract Signature {
    function callFoo(address addr, uint value) public returns (bool) {
        bytes memory data = abi.encodeWithSignature("foo(uint)", value);
        (bool status, ) = addr.call(data);
        return status;
    }
}