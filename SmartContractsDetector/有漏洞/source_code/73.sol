
browser
config
browser/Untitled.sol


0

[2] only remix transactions, script

Search transactions
remix.getFile(path): Returns the content of the file located at the given path

remix.setFile(path, content): set the content of the file located at the given path

remix.debug(hash): Start debugging a transaction.

remix.loadgist(id): Load a gist in the file explorer.

remix.loadurl(url): Load the given url in the file explorer. The url can be of type github, swarm, ipfs or raw http

remix.setproviderurl(url): Change the current provider to Web3 provider and set the url endpoint.

remix.execute(filepath): Run the script specified by file path. If filepath is empty, script currently displayed in the editor is executed.

remix.exeCurrent(): Run the script currently displayed in the editor

remix.help(): Display this help message

remix.debugHelp(): Display help message for debugging

 - Welcome to Remix v0.7.6 - 

You can use this terminal for: 
Checking transactions details and start debugging.
Running JavaScript scripts. The following libraries are accessible:
                      
web3 version 1.0.0
ethers.js
swarmgw
compilers - contains currently loaded compiler
Executing common command to interact with the Remix interface (see list of commands above). Note that these commands can also be included and run from a JavaScript script.
Use exports/.register(key, obj)/.remove(key)/.clear() to register and reuse object across script executions.
>
Compile
Run
Analysis
Testing
Debugger
Settings
Support
Current version:0.4.25-nightly.2018.5.16+commit.3897c367.Emscripten.clang


Auto compile

Enable Optimization

Hide warnings
Start to compile (Ctrl-S)

Swarm
Details
ABI
Bytecode
Static Analysis raised 5 warning(s) that requires your attention. Click here to show the warning(s).
Warning: This is a pre-release compiler version, please do not use it in production.
browser/Untitled.sol:2:1: Warning: Source file does not specify required compiler version!
contract wallet
^ (Relevant source part starts here and spans across multiple lines).
