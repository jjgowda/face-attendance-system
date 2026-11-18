(() => {
 const button=document.querySelector('#installApp'),dialog=document.querySelector('#installDialog');let pendingPrompt=null;
 const standalone=()=>window.matchMedia('(display-mode: standalone)').matches||navigator.standalone===true;
 const refresh=()=>{button.hidden=standalone();};refresh();
 window.addEventListener('beforeinstallprompt',event=>{event.preventDefault();pendingPrompt=event;refresh();});
 window.addEventListener('appinstalled',()=>{pendingPrompt=null;button.hidden=true;if(dialog.open)dialog.close();});
 window.matchMedia('(display-mode: standalone)').addEventListener('change',refresh);
 function instructions(){
  const ua=navigator.userAgent,ios=/iPad|iPhone|iPod/.test(ua)||(navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1);
  document.querySelector('#installInstructions').textContent=ios?'Open this website in Safari. Tap Share, choose Add to Home Screen, then tap Add.':/Android/.test(ua)?'Open this website in Chrome. Tap the three-dot menu, then Install app or Add to Home screen.':'Use the Install app option in your browser’s address bar or menu. In Safari on Mac, choose File → Add to Dock.';
  if(!dialog.open)dialog.showModal();
 }
 button.onclick=async()=>{
  if(!pendingPrompt){instructions();return;}
  const prompt=pendingPrompt;pendingPrompt=null;button.disabled=true;
  try{await prompt.prompt();const result=await prompt.userChoice;if(result.outcome==='accepted')button.hidden=true;}catch{instructions();}finally{button.disabled=false;}
 };
 document.querySelector('#closeInstall').onclick=()=>dialog.close();document.querySelector('#installDone').onclick=()=>dialog.close();
 if('serviceWorker' in navigator)window.addEventListener('load',()=>{navigator.serviceWorker.register('/sw.js').catch(()=>{});});
})();
