pragma solidity ^0.4.25;

contract MyContract {

    function withdraw() {
        if (msg.sender.call.value(1)()) {
        /*...*/
        }
    }
}