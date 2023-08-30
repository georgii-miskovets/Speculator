function t431_init(recid){var tableHead=t431__escapeHTML($('#rec'+recid+' .t431 .t431__data-part1').html()||"");var tableBody=t431__escapeHTML($('#rec'+recid+' .t431 .t431__data-part2').html()||"");var tableColSize=$('#rec'+recid+' .t431 .t431__table').attr("data-table-width");var hasTargetBlank=$('#rec'+recid+' .t431 .t431__table').attr("data-target-blank");var tHead=t431_parseData(tableHead);var tBody=t431_parseData(tableBody);var colSize=t431_parseData(tableColSize);var maxColNum=t431__findMaxRowLengthInTable(tHead,tBody);var colWidth=t431__setColumnsWidth(colSize,maxColNum,recid);var container=$('#rec'+recid+' .t431 .t431__table');var html="";if(tHead){html+=t431__generateTable(tHead,"th",hasTargetBlank,colWidth,maxColNum)}
if(tBody){html+=t431__generateTable(tBody,"td",hasTargetBlank,colWidth,maxColNum)}
container.append(html)}
function t431__findMaxRowLengthInTable(arrayHead,arrayData){var headMaxLength=0;var dataMaxLength=0;if(arrayHead){headMaxLength=t431__findMaxRowLengInArray(arrayHead)}
if(arrayData){dataMaxLength=t431__findMaxRowLengInArray(arrayData)}
if(dataMaxLength>headMaxLength){return dataMaxLength}else{return headMaxLength}}
function t431__escapeHTML(string){var html=string.replace(/&lt;/g,'<').replace(/&gt;/g,'>').replace(/&amp;/g,'&').replace(/&nbsp;/g,' ');var result="";var allowedTags="";['b','i','u','ul','li','ol','br','img','s','sub','sup','span','hr','pre','code','mark','strong','small'].forEach(function(value){allowedTags+=":not("+value+")"});var allowedAttrs=['alt','class','title','id','src','style','width','height','data-replace-key'];var fakeDOM=document.implementation.createHTMLDocument('fake');$.each($.parseHTML(html,fakeDOM)||[],function(i,$el){var el=$($el)[0];if(!$($el).is(allowedTags)){if(el.nodeType!==3&&el.nodeType!==8){var temp=document.createElement(el.tagName);allowedAttrs.forEach(function(value){if(el.getAttribute(value)!==null){temp.setAttribute(value,el.getAttribute(value).replace(/javascript:/gi,''))}});temp.textContent=el.textContent;result+=temp.outerHTML}else{result+=el.textContent}}});return result}
function t431__findMaxRowLengInArray(curArray){var maxLength=0;for(var i=0;i<curArray.length;i++){if(curArray[i].length>maxLength){maxLength=curArray[i].length}}
return maxLength}
function t431__setColumnsWidth(colWidth,colsNumber,recid){if(colWidth){return colWidth[0]}else{var tableWidth=$('#rec'+recid+' .t431 .t-container .t-col').width();return(tableWidth/colsNumber+"px")}}
function t431__generateTable(arrayValues,colTag,hasTargetBlank,colWidth,maxColNumber){var html="";var tag="";if(colTag=="td"){tag="tbody"}else{tag="thead"}
html+='<'+tag+' class="t431__'+tag+'">';for(var i=0;i<arrayValues.length;i++){if(colTag=="td"){if((i+1)%2>0){html+='<tr class="t431__oddrow">'}else{html+='<tr class="t431__evenrow">'}}else{html+='<tr>'}
var addingCols=0;if(arrayValues[i].length<maxColNumber){addingCols=maxColNumber-arrayValues[i].length}
for(var j=0;j<(arrayValues[i].length+addingCols);j++){if(arrayValues[i][j]){var curWidth="";if(Array.isArray(colWidth)&&colWidth[j]){curWidth=colWidth[j].myText}else{curWidth=colWidth}
var ColWithAttr='';if(colTag=="td"){ColWithAttr='<td class="t431__td t-text" width="'+curWidth+'">'}else{ColWithAttr='<th class="t431__th t-title" width="'+curWidth+'">'}
if(arrayValues[i][j].myHref){var tBlank="";if(hasTargetBlank){tBlank="target=\"_blank\""}
var linkWithAttr="";var linkCloseTag="";if(arrayValues[i][j].myHrefType=="link"){linkWithAttr='<a href="'+arrayValues[i][j].myHref+'"'+tBlank+'>';linkCloseTag='</a>'}else{linkWithAttr='<div class="t431__btnwrapper"><a href="'+arrayValues[i][j].myHref+'"'+tBlank+' class="t-btn t-btn_sm"><table style="width:100%; height:100%"><tr><td>';linkCloseTag='</td></tr></table></a></div>'}
html+=ColWithAttr+linkWithAttr+arrayValues[i][j].myText+linkCloseTag+'</'+colTag+'>'}else{html+=ColWithAttr+arrayValues[i][j].myText+'</'+colTag+'>'}}else{html+='<'+colTag+' class="t431__'+colTag+'" width="'+curWidth+'">'+'</'+colTag+'>'}}
html+="</tr>"}
html+="</"+tag+">";return html}
function t431_parseData(data){if(data!==""&&typeof data!="undefined"){data=t431__addBrTag(data);var arrayTable=[];var arrayRow=[];var curItem={myText:"",myHref:"",myHrefType:""};var hasLink="";var hasLinkWithSpace="";var hasBtn="";var hasBtnWithSpace="";var endLine="";for(var i=0;i<data.length;i++){if(data[i]==";"&&!(data.slice(i-4,i)=="&lt;"||data.slice(i-4,i)=="&gt;"||data.slice(i-5,i)=="&amp;"||data.slice(i-6,i)=="&nbsp;")){arrayRow.push(curItem);curItem={myText:"",myHref:""};hasLink="";hasLinkWithSpace="";hasBtn="";hasBtnWithSpace=""}else{if(hasLink=="link="||hasLinkWithSpace==" link="||hasBtn=="button="||hasBtnWithSpace==" button="){if(curItem.myHref===""&&hasLink==="link="){curItem.myText=curItem.myText.slice(0,-5);curItem.myHrefType="link"}else{if(curItem.myHref===""&&hasLinkWithSpace===" link="){curItem.myText=curItem.myText.slice(0,-6);curItem.myHrefType="link"}else{if(curItem.myHref===""&&hasBtn==="button="){curItem.myText=curItem.myText.slice(0,-7);curItem.myHrefType="btn"}else{if(curItem.myHref===""&&hasBtnWithSpace===" button="){curItem.myText=curItem.myText.slice(0,-8);curItem.myHrefType="btn"}}}}
curItem.myHref+=(data[i])}else{curItem.myText+=(data[i]);hasLink=t431__checkSubstr("link=",hasLink,data[i]);hasLinkWithSpace=t431__checkSubstr(" link=",hasLinkWithSpace,data[i]);hasBtn=t431__checkSubstr("button=",hasBtn,data[i]);hasBtnWithSpace=t431__checkSubstr(" button=",hasBtnWithSpace,data[i])}
endLine=t431__checkSubstr("<br />",endLine,data[i]);if(endLine=="<br />"){if(curItem.myHref){curItem.myHref=curItem.myHref.slice(0,-6)}else{curItem.myText=curItem.myText.slice(0,-6)}
arrayRow.push(curItem);arrayTable.push(arrayRow);curItem={myText:"",myHref:""};hasLink="";hasLinkWithSpace="";hasBtn="";hasBtnWithSpace="";arrayRow=[]}}}
if(arrayRow.length>0||curItem.myText!==""){if(curItem!==""){arrayRow.push(curItem)}
arrayTable.push(arrayRow)}}
return arrayTable}
function t431__checkSubstr(targetSubstr,curSubstr,curSymbol){if(!curSubstr&&curSymbol==targetSubstr[0]){return curSymbol}else{if(curSubstr){for(var i=0;i<(targetSubstr.length-1);i++){if(curSubstr[curSubstr.length-1]==targetSubstr[i]&&curSymbol==targetSubstr[i+1]){return(curSubstr+=curSymbol)}}}}}
function t431__addBrTag(oldStringItem){var newStringItem="";for(var i=0;i<oldStringItem.length;i++){if(oldStringItem[i]=="\n"||oldStringItem[i]=="\r"){newStringItem+="<br />"}else{newStringItem+=oldStringItem[i]}}
return newStringItem.replace(/&nbsp;/g,' ')}
function t431_createTable(recid,tablehead,tabledata,tablecolsize,hastargetblank,btnstyles,t431__tdstyles,t431__thstyles,t431__oddrowstyles,t431__evenrowstyles){var t431__arrayColSize=t431_parseData(tablecolsize);var t431__arrayHead=t431_parseData(tablehead);var t431__arrayData=t431_parseData(tabledata);var t431__maxcolnumber=t431__findMaxRowLengthInTable(t431__arrayHead,t431__arrayData);var t431__colWidth=t431__setColumnsWidth(t431__arrayColSize,t431__maxcolnumber,recid);if(t431__colWidth[0].myText&&t431__colWidth[0].myText[t431__colWidth[0].myText.length-1]=="%"){for(var i=0;i<t431__colWidth.length;i++){t431__colWidth[i].myText=t431__colWidth[i].myText.slice(0,-1);t431__colWidth[i].myText+="vw"}}
var t431__container=$('#rec'+recid+' .t431 .t-container .t431__table');var t431__htmlTable="";if(t431__arrayHead){t431__htmlTable+=t431__generateHtml(recid,t431__arrayHead,"th",hastargetblank,t431__colWidth,btnstyles,t431__thstyles,null,null,t431__maxcolnumber)}
t431__container.append(t431__htmlTable);t431__htmlTable="";if(t431__arrayData){t431__htmlTable+=t431__generateHtml(recid,t431__arrayData,"td",hastargetblank,t431__colWidth,btnstyles,t431__tdstyles,t431__oddrowstyles,t431__evenrowstyles,t431__maxcolnumber)}
t431__container.append(t431__htmlTable)}
function t431__generateHtml(recid,arrayValues,coltag,hastargetblank,colWidth,btnstyles,colstyles,oddrowstyles,evenrowstyles,maxcolnumber){var t431__htmlpart="";if(coltag=="td"){var t431__theadorbodytag="tbody"}else{var t431__theadorbodytag="thead"}
t431__htmlpart+='<'+t431__theadorbodytag+' class="t431__'+t431__theadorbodytag+'">';var t431__firstbodyrowstyle="";if($('#rec'+recid+' .t431 .t-container .t431__thead th').length>0&&$('#rec'+recid+' .t431 .t-container .t431__thead th').css("border-bottom-width")[0]!="0"){t431__firstbodyrowstyle="border-top: 0 !important;"}
for(var i=0;i<arrayValues.length;i++){if(coltag=="td"){if((i+1)%2>0){t431__htmlpart+="<tr class=\"t431__oddrow\""+"style=\""+oddrowstyles+"\">"}else{t431__htmlpart+="<tr class=\"t431__evenrow\""+"style=\""+evenrowstyles+"\">"}}else{t431__htmlpart+="<tr>"}
var t431__addingcols=0;if(arrayValues[i].length<maxcolnumber){t431__addingcols=maxcolnumber-arrayValues[i].length}
for(var j=0;j<(arrayValues[i].length+t431__addingcols);j++){if(arrayValues[i][j]){if(Array.isArray(colWidth)&&colWidth[j]){var t431__curWidth=colWidth[j].myText}else{var t431__curWidth=colWidth}
if(i==0&&coltag=="td"){var t431__colwithattr="<"+coltag+" class=\"t431__"+coltag+"\" style=\"width:"+t431__curWidth+";"+colstyles+t431__firstbodyrowstyle+"\">"}else{var t431__colwithattr="<"+coltag+" class=\"t431__"+coltag+"\" style=\"width:"+t431__curWidth+";"+colstyles+"\">"}
if(arrayValues[i][j].myHref){var t431__tblank="";if(hastargetblank){var t431__tblank="target=\"_blank\""}
if(arrayValues[i][j].myHrefType=="link"){var t431__linkwithattr="<a href=\""+arrayValues[i][j].myHref+"\""+t431__tblank+">";var t431__linkclosetag="</a>"}else{var t431__linkwithattr="<div class=\"t431__btnwrapper\"><a href=\""+arrayValues[i][j].myHref+"\""+t431__tblank+" class=\"t-btn t-btn_sm\" style=\""+btnstyles+"\"><table style=\"width:100%; height:100%;\"><tr><td>";var t431__linkclosetag="</td></tr></table></a></div>"}
t431__htmlpart+=t431__colwithattr+t431__linkwithattr+arrayValues[i][j].myText+t431__linkclosetag+"</"+coltag+">"}else{t431__htmlpart+=t431__colwithattr+arrayValues[i][j].myText+"</"+coltag+">"}}else{t431__htmlpart+="<"+coltag+" class=\"t431__"+coltag+"\" style=\"width:"+t431__curWidth+";"+colstyles+"\">"+"</"+coltag+">"}}
t431__htmlpart+="</tr>"}
t431__htmlpart+="</"+t431__theadorbodytag+">";return t431__htmlpart}
function t446_init(recid){var el=$('#rec'+recid);var mobile=el.find('.t446__mobile');var fixedBlock=mobile.css('position')==='fixed'&&mobile.css('display')==='block';setTimeout(function(){el.find('.t-menu__link-item:not(.t-menusub__target-link):not(.tooltipstered):not(.t794__tm-link)').on('click',function(){if($(this).is(".t-menu__link-item.tooltipstered, .t-menu__link-item.t-menusub__target-link, .t-menu__link-item.t794__tm-link, .t-menu__link-item.t966__tm-link, .t-menu__link-item.t978__tm-link")){return}
if(fixedBlock){mobile.trigger('click')}});el.find('.t-menusub__link-item').on('click',function(){if(fixedBlock){mobile.trigger('click')}})},500)}
function t446_setLogoPadding(recid){if($(window).width()>980){var t446__menu=$('#rec'+recid+' .t446');var t446__logo=t446__menu.find('.t446__logowrapper');var t446__leftpart=t446__menu.find('.t446__leftwrapper');var t446__rightpart=t446__menu.find('.t446__rightwrapper');t446__leftpart.css("padding-right",t446__logo.width()/2+50);t446__rightpart.css("padding-left",t446__logo.width()/2+50)}}
function t446_checkOverflow(recid,menuheight){var t446__menu=$('#rec'+recid+' .t446');var t446__rightwr=t446__menu.find('.t446__rightwrapper');var t446__rightmenuwr=t446__rightwr.find('.t446__rightmenuwrapper');var t446__rightadditionalwr=t446__rightwr.find('.t446__additionalwrapper');var t446__burgeroverflow=t446__rightwr.find('.t446__burgerwrapper_overflow');var t446__burgerwithoutoverflow=t446__rightwr.find('.t446__burgerwrapper_withoutoverflow');if(menuheight>0){var t446__height=menuheight}else{var t446__height=80}
if($(window).width()>980&&(t446__rightmenuwr.width()+t446__rightadditionalwr.width())>t446__rightwr.width()){t446__menu.css("height",t446__height*2);t446__rightadditionalwr.css("float","right");t446__burgeroverflow.css("display","table-cell");t446__burgerwithoutoverflow.css("display","none")}else{if(t446__menu.height()>t446__height){t446__menu.css("height",t446__height)}
if(t446__rightadditionalwr.css("float")=="right"){t446__rightadditionalwr.css("float","none")}
t446__burgeroverflow.css("display","none");t446__burgerwithoutoverflow.css("display","table-cell")}}
function t446_highlight(){var url=window.location.href;var pathname=window.location.pathname;if(url.substr(url.length-1)=="/"){url=url.slice(0,-1)}
if(pathname.substr(pathname.length-1)=="/"){pathname=pathname.slice(0,-1)}
if(pathname.charAt(0)=="/"){pathname=pathname.slice(1)}
if(pathname==""){pathname="/"}
$(".t446__list_item a[href='"+url+"']").addClass("t-active");$(".t446__list_item a[href='"+url+"/']").addClass("t-active");$(".t446__list_item a[href='"+pathname+"']").addClass("t-active");$(".t446__list_item a[href='/"+pathname+"']").addClass("t-active");$(".t446__list_item a[href='"+pathname+"/']").addClass("t-active");$(".t446__list_item a[href='/"+pathname+"/']").addClass("t-active")}
function t446_checkAnchorLinks(recid){if($(window).width()>=960){var t446_navLinks=$("#rec"+recid+" .t446__list_item a:not(.tooltipstered)[href*='#']");if(t446_navLinks.length>0){t446_catchScroll(t446_navLinks)}}}
function t446_catchScroll(t446_navLinks){var t446_clickedSectionId=null,t446_sections=new Array(),t446_sectionIdTonavigationLink=[],t446_interval=100,t446_lastCall,t446_timeoutId;t446_navLinks=$(t446_navLinks.get().reverse());t446_navLinks.each(function(){var t446_cursection=t446_getSectionByHref($(this));if(typeof t446_cursection.attr("id")!="undefined"){t446_sections.push(t446_cursection)}
t446_sectionIdTonavigationLink[t446_cursection.attr("id")]=$(this)});t446_updateSectionsOffsets(t446_sections);t446_sections.sort(function(a,b){return b.attr("data-offset-top")-a.attr("data-offset-top")});$(window).bind('resize',t_throttle(function(){t446_updateSectionsOffsets(t446_sections)},200));$('.t446').bind('displayChanged',function(){t446_updateSectionsOffsets(t446_sections)});setInterval(function(){t446_updateSectionsOffsets(t446_sections)},5000);t446_highlightNavLinks(t446_navLinks,t446_sections,t446_sectionIdTonavigationLink,t446_clickedSectionId);t446_navLinks.click(function(){var t446_clickedSection=t446_getSectionByHref($(this));if(!$(this).hasClass("tooltipstered")&&typeof t446_clickedSection.attr("id")!="undefined"){t446_navLinks.removeClass('t-active');$(this).addClass('t-active');t446_clickedSectionId=t446_getSectionByHref($(this)).attr("id")}});$(window).scroll(function(){var t446_now=new Date().getTime();if(t446_lastCall&&t446_now<(t446_lastCall+t446_interval)){clearTimeout(t446_timeoutId);t446_timeoutId=setTimeout(function(){t446_lastCall=t446_now;t446_clickedSectionId=t446_highlightNavLinks(t446_navLinks,t446_sections,t446_sectionIdTonavigationLink,t446_clickedSectionId)},t446_interval-(t446_now-t446_lastCall))}else{t446_lastCall=t446_now;t446_clickedSectionId=t446_highlightNavLinks(t446_navLinks,t446_sections,t446_sectionIdTonavigationLink,t446_clickedSectionId)}})}
function t446_updateSectionsOffsets(sections){$(sections).each(function(){var t446_curSection=$(this);t446_curSection.attr("data-offset-top",t446_curSection.offset().top)})}
function t446_getSectionByHref(curlink){var t446_curLinkValue=curlink.attr("href").replace(/\s+/g,'');if(t446_curLinkValue[0]=='/'){t446_curLinkValue=t446_curLinkValue.substring(1)}
if(curlink.is('[href*="#rec"]')){return $(".r[id='"+t446_curLinkValue.substring(1)+"']")}else{return $(".r[data-record-type='215']").has("a[name='"+t446_curLinkValue.substring(1)+"']")}}
function t446_highlightNavLinks(t446_navLinks,t446_sections,t446_sectionIdTonavigationLink,t446_clickedSectionId){var t446_scrollPosition=$(window).scrollTop(),t446_valueToReturn=t446_clickedSectionId;if(t446_sections.length!=0&&t446_clickedSectionId==null&&t446_sections[t446_sections.length-1].attr("data-offset-top")>(t446_scrollPosition+300)){t446_navLinks.removeClass('t-active');return null}
$(t446_sections).each(function(e){var t446_curSection=$(this),t446_sectionTop=t446_curSection.attr("data-offset-top"),t446_id=t446_curSection.attr('id'),t446_navLink=t446_sectionIdTonavigationLink[t446_id];if(((t446_scrollPosition+300)>=t446_sectionTop)||(t446_sections[0].attr("id")==t446_id&&t446_scrollPosition>=$(document).height()-$(window).height())){if(t446_clickedSectionId==null&&!t446_navLink.hasClass('t-active')){t446_navLinks.removeClass('t-active');t446_navLink.addClass('t-active');t446_valueToReturn=null}else{if(t446_clickedSectionId!=null&&t446_id==t446_clickedSectionId){t446_valueToReturn=null}}
return!1}});return t446_valueToReturn}
function t446_setPath(){}
function t446_setBg(recid){var window_width=$(window).width();if(window_width>980){$(".t446").each(function(){var el=$(this);if(el.attr('data-bgcolor-setbyscript')=="yes"){var bgcolor=el.attr("data-bgcolor-rgba");el.css("background-color",bgcolor)}})}else{$(".t446").each(function(){var el=$(this);var bgcolor=el.attr("data-bgcolor-hex");el.css("background-color",bgcolor);el.attr("data-bgcolor-setbyscript","yes")})}}
function t446_appearMenu(recid){var window_width=$(window).width();if(window_width>980){$(".t446").each(function(){var el=$(this);var appearoffset=el.attr("data-appearoffset");if(appearoffset!=""){if(appearoffset.indexOf('vh')>-1){appearoffset=Math.floor((window.innerHeight*(parseInt(appearoffset)/100)))}
appearoffset=parseInt(appearoffset,10);if($(window).scrollTop()>=appearoffset){if(el.css('visibility')=='hidden'){el.finish();el.css("top","-50px");el.css("visibility","visible");el.animate({"opacity":"1","top":"0px"},200,function(){})}}else{el.stop();el.css("visibility","hidden")}}})}}
function t446_changebgopacitymenu(recid){var window_width=$(window).width();if(window_width>980){$(".t446").each(function(){var el=$(this);var bgcolor=el.attr("data-bgcolor-rgba");var bgcolor_afterscroll=el.attr("data-bgcolor-rgba-afterscroll");var bgopacityone=el.attr("data-bgopacity");var bgopacitytwo=el.attr("data-bgopacity-two");var menushadow=el.attr("data-menushadow");if(menushadow=='100'){var menushadowvalue=menushadow}else{var menushadowvalue='0.'+menushadow}
if($(window).scrollTop()>20){el.css("background-color",bgcolor_afterscroll);if(bgopacitytwo=='0'||menushadow==' '){el.css("box-shadow","none")}else{el.css("box-shadow","0px 1px 3px rgba(0,0,0,"+menushadowvalue+")")}}else{el.css("background-color",bgcolor);if(bgopacityone=='0.0'||menushadow==' '){el.css("box-shadow","none")}else{el.css("box-shadow","0px 1px 3px rgba(0,0,0,"+menushadowvalue+")")}}})}}
function t446_createMobileMenu(recid){var window_width=$(window).width(),el=$("#rec"+recid),menu=el.find(".t446"),burger=el.find(".t446__mobile");if(menu.hasClass('t446__mobile_burgerhook')){burger.find('.t446__mobile_burger').wrap('<a href="#menuopen"></a>')}else{burger.click(function(e){menu.fadeToggle(300);$(this).toggleClass("t446_opened")})}
$(window).bind('resize',t_throttle(function(){window_width=$(window).width();if(window_width>980){menu.fadeIn(0)}},200));el.find('.t-menu__link-item').on('click',function(){if(!$(this).hasClass('t966__tm-link')&&!$(this).hasClass('t978__tm-link')){t446_hideMenuOnMobile($(this),el)}});el.find('.t446__logowrapper2 a').on('click',function(){t446_hideMenuOnMobile($(this),el)})}
function t446_hideMenuOnMobile($this,el){if($(window).width()<960){var url=$this.attr('href').trim();var menu=el.find('.t446');var burger=el.find('.t446__mobile');if(url.length&&url[0]==='#'){burger.removeClass('t446_opened');if(menu.is('.t446__positionabsolute')){menu.fadeOut(0)}else{menu.fadeOut(300)}
return!0}}}
function t734_init(recid){var rec=$('#rec'+recid);if($('body').find('.t830').length>0){if(rec.find('.t-slds__items-wrapper').hasClass('t-slds_animated-none')){t_onFuncLoad('t_sldsInit',function(){t_sldsInit(recid)})}else{setTimeout(function(){t_onFuncLoad('t_sldsInit',function(){t_sldsInit(recid)})},500)}}else{t_onFuncLoad('t_sldsInit',function(){t_sldsInit(recid)})}
rec.find('.t734').bind('displayChanged',function(){t_onFuncLoad('t_slds_updateSlider',function(){t_slds_updateSlider(recid)})})}