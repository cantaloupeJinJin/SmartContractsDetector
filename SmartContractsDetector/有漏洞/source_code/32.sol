pragma solidity ^0.4.25;

contract Victim {
    function withdraw() {
        if (msg.sender.call.value()()) {
        }
    }
}
